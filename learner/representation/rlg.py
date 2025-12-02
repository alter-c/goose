from .base_class import *
from planning.translate.pddl import Atom, NegatedAtom, Truth
from util.llm import LLM
import math


class RLG_FEATURES(Enum):
    P = 0  # is predicate
    A = 1  # is action
    G = 2  # is positive goal (grounded)
    N = 3  # is negative goal (grounded)
    S = 4  # is activated (grounded)
    O = 5  # is object

RLG_EDGE_LABELS = OrderedDict(
    {
        "neutral": 0,
        "ground": 1,
        "pre_pos": 2,
        "pre_neg": 3,
        "eff_pos": 4,
        "eff_neg": 5,
    }
)

ENC_FEAT_SIZE = len(RLG_FEATURES)
VAR_FEAT_SIZE = 4
INDEX_FEAT_SIZE = 4
LLM_FEAT_SIZE = 4
HEURISTIC_FEAT_SIZE = 1

class RichLearningGraph(Representation, ABC):
    name = "rlg"
    feature_size_list = [ENC_FEAT_SIZE, VAR_FEAT_SIZE, INDEX_FEAT_SIZE, LLM_FEAT_SIZE,HEURISTIC_FEAT_SIZE]
    n_node_features = sum(feature_size_list)
    n_edge_labels = len(RLG_EDGE_LABELS)
    directed = False
    lifted = True

    def __init__(self, domain_pddl: str, problem_pddl: str):
        super().__init__(domain_pddl, problem_pddl)
        
    def detect_goal_size(self):
        return len(self.problem.goal.parts)

    def _construct_if(self) -> None:
        """Precompute a seeded randomly generated injective index function"""
        self._if = []
        image = set()  # check injectiveness

        # TODO read max range from problem and lazily compute
        for idx in range(60):
            torch.manual_seed(idx)  # fixed seed for reproducibility
            rep = 2 * torch.rand(VAR_FEAT_SIZE) - 1  # U[-1,1]
            rep /= torch.linalg.norm(rep)
            self._if.append(rep)
            key = tuple(rep.tolist())
            assert key not in image
            image.add(key)
        return

    def _construct_index(self) -> None:
        """Precompute index for objects"""

        self._index = []
        index_image = set()
        
        MAX_INDEX = 200
        for idx in range(MAX_INDEX):
            torch.manual_seed(idx)
            rep = 2 * torch.rand(INDEX_FEAT_SIZE) - 1  # U[-1,1]
            rep /= torch.linalg.norm(rep)
            self._index.append(rep)
            key = tuple(rep.tolist())
            assert key not in index_image
            index_image.add(key)
        return

    def _feature(self, node_type: RLG_FEATURES) -> Tensor:
        ret = torch.zeros(self.n_node_features)
        ret[node_type.value] = 1
        return ret

    def _if_feature(self, idx: int) -> Tensor:
        ret = torch.zeros(self.n_node_features)
        ret[ENC_FEAT_SIZE: ENC_FEAT_SIZE + VAR_FEAT_SIZE] = self._if[idx]
        return ret

    def _index_feature(self, idx: int) -> Tensor:
        ret = torch.zeros(self.n_node_features)
        ret[ENC_FEAT_SIZE + VAR_FEAT_SIZE: ENC_FEAT_SIZE + VAR_FEAT_SIZE + INDEX_FEAT_SIZE] = self._index[idx]
        return ret

    def _llm_feature(self, embedding: str) -> Tensor:
        ret = torch.zeros(self.n_node_features)
        ret[-(LLM_FEAT_SIZE + HEURISTIC_FEAT_SIZE): -HEURISTIC_FEAT_SIZE] = torch.Tensor(embedding)
        return ret
    
    def _distance_feature(self, distance: int) -> Tensor:
        ret = torch.zeros(self.n_node_features)
        ret[-HEURISTIC_FEAT_SIZE:] = torch.Tensor([distance])
        return ret

    def _compute_graph_representation(self) -> None:
        """TODO: reference definition of this graph representation"""

        self._construct_if()
        self._construct_index()
        self.llm = LLM(self.problem, self.domain_pddl, self.problem_pddl, feat_size=LLM_FEAT_SIZE)
        self.llm_embeddings = self.llm.embedding()

        G = self._create_graph()

        # add distance node
        self.goal_size = self.detect_goal_size()
        if "n-puzzle" in self.problem.domain_name:
            for i in range(self.goal_size):
                distance_node = (f"distance-{i}") 
                G.add_node(distance_node, x=self._distance_feature(i))
        else:
            G.add_node("distance-0", x=self._distance_feature(0))

        # objects
        for i, obj in enumerate(self.problem.objects):
            G.add_node(obj.name, x=self._feature(RLG_FEATURES.O))  # add object node

        index = 0
        # predicates
        largest_predicate = 0
        for pred in self.problem.predicates:
            largest_predicate = max(largest_predicate, len(pred.arguments))
            pred_embedding = self.llm_embeddings[pred.name]
            G.add_node(pred.name, x=self._feature(RLG_FEATURES.P) + self._index_feature(index) + self._llm_feature(pred_embedding))  # add predicate node
            index += 1

        # fully connected between objects and predicates
        for pred in self.problem.predicates:
            for obj in self.problem.objects:
                G.add_edge(
                    u_of_edge=pred.name, v_of_edge=obj.name, edge_label=RLG_EDGE_LABELS["neutral"]
                )

        # goal (state gets dealt with in state_to_tensor)
        if len(self.problem.goal.parts) == 0:
            goals = [self.problem.goal]
        else:
            goals = self.problem.goal.parts
        for fact in goals:
            assert type(fact) in {Atom, NegatedAtom}

            # may have negative goals
            is_negated = type(fact) == NegatedAtom

            pred = fact.predicate
            args = fact.args
            goal_node = (pred, args)

            if is_negated:
                x = self._feature(RLG_FEATURES.N)
                self._neg_goal_nodes.add(goal_node)
            else:
                x = self._feature(RLG_FEATURES.G)
                self._pos_goal_nodes.add(goal_node)
            G.add_node(goal_node, x=x)  # add grounded predicate node

            for i, arg in enumerate(args):
                goal_var_node = (goal_node, i)
                G.add_node(goal_var_node, x=self._if_feature(idx=i))

                # connect variable to predicate
                G.add_edge(
                    u_of_edge=goal_node,
                    v_of_edge=goal_var_node,
                    edge_label=RLG_EDGE_LABELS["ground"],
                )

                # connect variable to object
                assert arg in G.nodes()
                G.add_edge(
                    u_of_edge=goal_var_node, v_of_edge=arg, edge_label=RLG_EDGE_LABELS["ground"]
                )

                # add extra directed edges
                # connect object to predicate
                G.add_edge(
                    u_of_edge=goal_node, v_of_edge=arg, edge_label=RLG_EDGE_LABELS["ground"]
                )

            # connect grounded fact to predicate
            assert pred in G.nodes()
            G.add_edge(u_of_edge=goal_node, v_of_edge=pred, edge_label=RLG_EDGE_LABELS["ground"])
        # end goal

        # actions
        largest_action_schema = 0
        for action in self.problem.actions:
            action_embedding = self.llm_embeddings[action.name]
            G.add_node(action.name, x=self._feature(RLG_FEATURES.A) + self._index_feature(index) + self._llm_feature(action_embedding))
            index += 1
            action_args = {}

            largest_action_schema = max(largest_action_schema, len(action.parameters))
            for i, arg in enumerate(action.parameters):
                arg_node = (action.name, f"action-var-{i}")  # action var
                G.add_node(arg_node, x=self._if_feature(idx=i))
                action_args[arg.name] = arg_node
                G.add_edge(
                    u_of_edge=action.name, v_of_edge=arg_node, edge_label=RLG_EDGE_LABELS["neutral"]
                )

            def deal_with_action_prec_or_eff(predicates, edge_label):
                for z, predicate in enumerate(predicates):
                    pred = predicate.predicate
                    aux_node = (pred, f"{edge_label}-aux-{z}")  # aux node for duplicate preds
                    G.add_node(aux_node, x=self._zero_node())

                    assert pred in G.nodes()
                    G.add_edge(
                        u_of_edge=pred, v_of_edge=aux_node, edge_label=RLG_EDGE_LABELS[edge_label]
                    )

                    if len(predicate.args) > 0:
                        for j, arg in enumerate(predicate.args):
                            prec_arg_node = (arg, f"{edge_label}-aux-{z}-var-{j}")  # aux var
                            G.add_node(prec_arg_node, x=self._if_feature(idx=j))
                            G.add_edge(
                                u_of_edge=aux_node,
                                v_of_edge=prec_arg_node,
                                edge_label=RLG_EDGE_LABELS[edge_label],
                            )

                            if arg in action_args:
                                action_arg_node = action_args[arg]
                                G.add_edge(
                                    u_of_edge=prec_arg_node,
                                    v_of_edge=action_arg_node,
                                    edge_label=RLG_EDGE_LABELS[edge_label],
                                )
                    else:  # unitary predicate so connect directly to action
                        G.add_edge(
                            u_of_edge=aux_node,
                            v_of_edge=action.name,
                            edge_label=RLG_EDGE_LABELS[edge_label],
                        )
                return

            pos_pres = [p for p in action.precondition.parts if type(p) == Atom]
            neg_pres = [p for p in action.precondition.parts if type(p) == NegatedAtom]
            pos_effs = [p.literal for p in action.effects if type(p.literal) == Atom]
            neg_effs = [p.literal for p in action.effects if type(p.literal) == NegatedAtom]
            for p in action.effects:
                assert type(p.condition) == Truth  # no conditional effects
            deal_with_action_prec_or_eff(pos_pres, "pre_pos")
            deal_with_action_prec_or_eff(neg_pres, "pre_neg")
            deal_with_action_prec_or_eff(pos_effs, "eff_pos")
            deal_with_action_prec_or_eff(neg_effs, "eff_neg")
        # end actions

        assert largest_predicate > 0
        assert largest_action_schema > 0

        # map node name to index
        self._node_to_i = {}
        for i, node in enumerate(G.nodes):
            self._node_to_i[node] = i
        self.G = G

        return

    def str_to_state(self, s) -> List[Tuple[str, List[str]]]:
        """Used in dataset construction to convert string representation of facts into a (pred, [args]) representation"""
        state = []
        for fact in s:
            fact = fact.replace(")", "").replace("(", "")
            toks = fact.split()
            if toks[0] == "=":
                continue
            if len(toks) > 1:
                state.append((toks[0], toks[1:]))
            else:
                state.append((toks[0], ()))
        return state

    def state_to_tensor(self, state: List[Tuple[str, List[str]]]) -> TGraph:
        """States are represented as a list of (pred, [args])"""
        x = self.x.clone()
        edge_indices = self.edge_indices.copy()
        i = len(x)

        to_add = sum(len(fact[1]) + 1 for fact in state)
        x = torch.nn.functional.pad(x, (0, 0, 0, to_add), "constant", 0)
        append_edge_index = []

        for fact in state:
            pred = fact[0]
            args = fact[1]
            if len(pred) == 0:
                continue  # somehow we get ("", []) facts???

            node = (pred, tuple(args))

            # activated proposition overlaps with a goal Atom or NegatedAtom
            if node in self._node_to_i:
                x[self._node_to_i[node]][RLG_FEATURES.S.value] = 1
                append_edge_index.append((self._node_to_i[node], self._node_to_i["distance-0"]))
                append_edge_index.append((self._node_to_i["distance-0"], self._node_to_i[node]))
                continue

            # activated proposition does not overlap with a goal
            true_node_i = i
            x[i][RLG_FEATURES.S.value] = 1
            i += 1

            # connect fact to predicate
            append_edge_index.append((true_node_i, self._node_to_i[pred]))
            append_edge_index.append((self._node_to_i[pred], true_node_i))

            # connect to predicates and objects
            for k, arg in enumerate(args):
                true_var_node_i = i
                x[i][ENC_FEAT_SIZE: ENC_FEAT_SIZE + VAR_FEAT_SIZE] = self._if[k]
                i += 1

                # connect variable to predicate
                append_edge_index.append((true_node_i, true_var_node_i))
                append_edge_index.append((true_var_node_i, true_node_i))

                # connect variable to object
                append_edge_index.append((true_var_node_i, self._node_to_i[arg]))
                append_edge_index.append((self._node_to_i[arg], true_var_node_i))

                # add extra directed edges
                # connect object to predicate
                append_edge_index.append((true_node_i, self._node_to_i[arg]))
                append_edge_index.append((self._node_to_i[arg], true_node_i))

            # connect state - distance - goal
            if "n-puzzle" in self.problem.domain_name and pred == "at":
                problem_size = int(math.sqrt(self.goal_size+1))
                tile_str, pos_str = args
                tile = int(tile_str.split('_')[1])
                _, pos_x, pos_y = pos_str.split('_')
                pos_x, pos_y = int(pos_x), int(pos_y)
                target_x = (tile-1) // problem_size + 1
                target_y = tile % problem_size if tile % problem_size != 0 else problem_size

                # connect state to distance
                distance = abs(pos_x - target_x) + abs(pos_y - target_y)
                distance_node = (f"distance-{distance}")
                append_edge_index.append((true_node_i, self._node_to_i[distance_node]))
                append_edge_index.append((self._node_to_i[distance_node], true_node_i))

                # connect distance to goal
                target_pos = f"p_{target_x}_{target_y}"
                goal_node = (pred, (tile_str, target_pos))
                append_edge_index.append((self._node_to_i[distance_node], self._node_to_i[goal_node]))
                append_edge_index.append((self._node_to_i[goal_node], self._node_to_i[distance_node]))


        edge_indices[RLG_EDGE_LABELS["ground"]] = torch.hstack(
            (edge_indices[RLG_EDGE_LABELS["ground"]], torch.tensor(append_edge_index).T)
        ).long()

        return x, edge_indices

    def state_to_cgraph(self, state: List[Tuple[str, List[str]]]) -> CGraph:
        """States are represented as a list of (pred, [args])"""
        c_graph = self.c_graph.copy()

        for fact in state:
            pred = fact[0]
            args = fact[1]

            node = (pred, tuple(args))

            # activated proposition overlaps with a goal Atom or NegatedAtom
            if node in self._pos_goal_nodes:
                c_graph.nodes[node]["colour"] = (
                    c_graph.nodes[node]["colour"] + ACTIVATED_POS_GOAL_COLOUR_SUFFIX
                )
                continue
            elif node in self._neg_goal_nodes:
                c_graph.nodes[node]["colour"] = (
                    c_graph.nodes[node]["colour"] + ACTIVATED_NEG_GOAL_COLOUR_SUFFIX
                )
                continue

            # else add node and corresponding edges to graph
            c_graph.add_node(node, colour=ACTIVATED_COLOUR)

            # connect fact to predicate
            c_graph.add_edge(u_of_edge=node, v_of_edge=pred, edge_label=RLG_EDGE_LABELS["ground"])
            c_graph.add_edge(v_of_edge=node, u_of_edge=pred, edge_label=RLG_EDGE_LABELS["ground"])

            # connect to predicates and objects
            for k, arg in enumerate(args):
                arg_node = (node, f"true-var-{k}")
                c_graph.add_node(arg_node, colour=str(k) + IF_COLOUR_SUFFIX)

                # connect variable to predicate
                c_graph.add_edge(
                    u_of_edge=node, v_of_edge=arg_node, edge_label=RLG_EDGE_LABELS["ground"]
                )
                c_graph.add_edge(
                    v_of_edge=node, u_of_edge=arg_node, edge_label=RLG_EDGE_LABELS["ground"]
                )

                # connect variable to object
                c_graph.add_edge(
                    u_of_edge=arg_node, v_of_edge=arg, edge_label=RLG_EDGE_LABELS["ground"]
                )
                c_graph.add_edge(
                    v_of_edge=arg_node, u_of_edge=arg, edge_label=RLG_EDGE_LABELS["ground"]
                )

        return c_graph
