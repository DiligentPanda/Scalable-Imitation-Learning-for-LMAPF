from light_malib.envs.LMAPF.map import Map, MapManager
import yaml

maps_path = "lmapf_lib/learn-to-follow/env/test-maps_mazes.yaml"

map_manager=MapManager()

map_manager.load_learn_to_follow_maps(maps_path)

print("#maps: ",len(map_manager))

map_manager[0].print_graph()