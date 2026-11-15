"""solfil: instance segmentation of solar filaments in GONG H-alpha full-disk images.

Inference code of the pipeline submitted to the Solar Filament Segmentation Challenge 2026 (team Hibiki8739).
`solfil.pipeline.run` turns a folder of test images into the submission file using the trained weights in
`weights/`; the code that trained every stage is in `training/`.
"""
__version__ = '1.0.0'
