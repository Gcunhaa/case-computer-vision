import fire
from .tracking_controller import TrackingController

def main():
    tracking_controller = TrackingController()
    fire.Fire(tracking_controller.process_video)

if __name__ == '__main__':
    main() 