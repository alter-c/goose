import os

if __name__ == "__main__":
    for planner in ["downward", "powerlifted"]:
        os.system(f"cd planners/{planner} && python3 build.py")
