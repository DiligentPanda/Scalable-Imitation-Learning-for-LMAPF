ROOT="../.."
MAP_PATH="example_problems/random.domain/maps/random-32-32-20.map"
# MAP_PATH="example_problems/warehouse.domain/maps/warehouse_small.map"
# MAP_PATH="example_problems/warehouse.domain/maps/warehouse_large.map"
# MAP_PATH="example_problems/warehouse.domain/maps/sortation_large.map"
# MAP_PATH="example_problems/city.domain/maps/Paris_1_256.map"
# MAP_PATH="example_problems/game.domain/maps/brc202d.map"
PLAN_PATH="test.json"

cd viz/script

python plan_viz.py --map $ROOT/$MAP_PATH --plan $ROOT/$PLAN_PATH --grid --aid --tid --static --ca