import os
import os.path as osp
SCRIPT_DIR = osp.dirname(osp.realpath(__file__))
REPO_ROOT = osp.dirname(osp.dirname(SCRIPT_DIR))
# path to your own image root, or set up DATAS same like me.
MID_JOURNEY_V6_IMAGE_ROOT=os.environ.get("MID_JOURNEY_V6_IMAGE_ROOT", osp.join(REPO_ROOT, "DATAS/midjourney-v6-images"))


