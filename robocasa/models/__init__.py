import os

assets_root = os.environ.get(
	"ROBOCASA_ASSETS_ROOT", os.path.join(os.path.dirname(__file__), "assets")
)
