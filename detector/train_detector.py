r"""
Train the brass note detector (Ultralytics YOLO) on a dataset made by make_dataset.py.

Every Ultralytics image augmentation is switched off: flips, rotation, scaling,
translation, mosaic/mixup and especially the color (HSV) changes, since the
image "colors" are HCQT harmonics and the rows are pitches. The data was already
augmented the musical way (stem subsets + pitch shift) when it was generated.

Run in the yolo environment:
  quick smoke test on the sample dataset (a few minutes):
    python R:\code\detector\train_detector.py --data sample --epochs 3 --name smoke
  real training, keeping weights every 5 epochs:
    python R:\code\detector\train_detector.py --data full_p9m9 --epochs 20 --save-period 5 --name full_p9m9_v1
Results go to R:\experiments\detector_runs\<name> (weights\best.pt, plots, results.csv).
"""
import argparse
import os

DATA_ROOT = r'R:\datasets\detector'
RUNS = r'R:\experiments\detector_runs'

# every augmentation setting Ultralytics might have; only the ones this version knows are used
AUGMENTATIONS_OFF = dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0,
                         shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0, bgr=0.0, mosaic=0.0,
                         mixup=0.0, cutmix=0.0, copy_paste=0.0, close_mosaic=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='dataset name under R:\\datasets\\detector')
    ap.add_argument('--name', required=True, help='name for this training run')
    ap.add_argument('--model', default='yolo11n.pt', help='pretrained weights to start from')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save-period', type=int, default=-1,
                    help='also keep weights every N epochs (epochN.pt), to score on real recordings later')
    a = ap.parse_args()

    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    known = vars(get_cfg())
    off = {k: v for k, v in AUGMENTATIONS_OFF.items() if k in known}
    skipped = sorted(set(AUGMENTATIONS_OFF) - set(off))
    print('Augmentations switched off:', ', '.join(sorted(off)))
    if skipped:
        print('(not in this Ultralytics version, nothing to switch off:', ', '.join(skipped) + ')')

    model = YOLO(a.model)
    model.train(data=os.path.join(DATA_ROOT, a.data, 'data.yaml'), epochs=a.epochs, imgsz=640,
                batch=a.batch, workers=a.workers, seed=a.seed, project=RUNS, name=a.name,
                exist_ok=False, plots=True, save_period=a.save_period, **off)
    print(f'Results in {os.path.join(RUNS, a.name)}')
    print('TRAINING DONE')


if __name__ == '__main__':
    main()
