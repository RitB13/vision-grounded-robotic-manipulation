from typing import Any, Dict, List

import pybullet


class WorldStateBuilder:
    """
    Converts the current PyBullet scene into a compact,
    JSON-serializable symbolic world state.

    This class reads simulator state but does not control
    or modify the robot.
    """

    def __init__(self, env):
        self.env = env

    def build(self) -> Dict[str, Any]:
        """
        Build the current world state.

        Returns
        -------
        dict
            JSON-serializable description of the objects
            currently present in the environment.
        """

        if not hasattr(self.env, "obj_name_to_id"):
            raise AttributeError(
                "Environment does not expose 'obj_name_to_id'."
            )

        objects: List[Dict[str, Any]] = []

        for name, body_id in self.env.obj_name_to_id.items():

            position, orientation = pybullet.getBasePositionAndOrientation(
                body_id
            )

            object_type = self._get_object_type(name)
            color = self._get_object_color(name)

            objects.append(
                {
                    "name": name,
                    "type": object_type,
                    "color": color,
                    "position": [
                        float(position[0]),
                        float(position[1]),
                        float(position[2]),
                    ],
                }
            )

        return {
            "objects": objects,
            "object_names": [
                obj["name"]
                for obj in objects
            ],
        }

    @staticmethod
    def _get_object_type(name: str) -> str:
        """
        Extract the object type from names such as:

            blue block
            green bowl
            red sphere
        """

        parts = name.lower().split()

        if len(parts) >= 2:
            return parts[1]

        return "unknown"

    @staticmethod
    def _get_object_color(name: str) -> str:
        """
        Extract the color from names such as:

            blue block
            green bowl
            red sphere
        """

        parts = name.lower().split()

        if len(parts) >= 1:
            return parts[0]

        return "unknown"


if __name__ == "__main__":
    # Standalone smoke test.
    #
    # This creates the same basic scene used by the current
    # CLIPort baseline and prints the generated world state.

    from .environment import PickPlaceEnv

    config = {
        "pick": [
            "blue block",
            "green block",
            "red block",
        ],
        "place": [],
    }

    env = PickPlaceEnv(gui=False)

    try:
        env.reset(config)

        builder = WorldStateBuilder(env)

        world_state = builder.build()

        print("\n" + "=" * 60)
        print("WORLD STATE")
        print("=" * 60)

        for obj in world_state["objects"]:
            print(
                f"\nObject: {obj['name']}"
                f"\n  Type:     {obj['type']}"
                f"\n  Color:    {obj['color']}"
                f"\n  Position: {obj['position']}"
            )

        print("\nObject names:")
        print(world_state["object_names"])

    finally:
        env.close()