export DEMO_VIDEO_DIR=~/Videos/$(date +%s)
mkdir -p "$DEMO_VIDEO_DIR"

GOAL_INDEX=0 dora run demo.yml
GOAL_INDEX=1 dora run demo.yml
GOAL_INDEX=2 dora run demo.yml
