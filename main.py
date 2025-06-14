from surveillance.video_processor import VideoProcessor

def main():
    model_paths = {
        'fire': 'models/fire_detection_model.h5',
        'mask': 'models/mask_detection_model.h5',
        'weapon': 'models/weapon_detection_model.h5',
        'fight': 'models/fight_detection_model.h5'
    }
    processor = VideoProcessor(model_paths)
    processor.process_stream()

if __name__ == '__main__':
    main()