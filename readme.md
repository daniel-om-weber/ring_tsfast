# Dependencies
1) `pip install -U "jax[cuda12]"`

2) `pip install imt-ring imt-diodem fire`

3) `pip install git+https://github.com/simon-bachhuber/imt_benchmark`

4) `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### Bug: `scipy.sparse` throws `CircularImportError`

1) `pip uninstall scipy` 

2) `conda install scipy`

# Retraining of `RING` -- Newer Version

1) `python train_step1_generateData_v2.py 65536 ring_data --mot-art --dof-configuration "['111']"`

2) `python train_step2_trainRing_v2.py ring_data 512 4800 --drop-dof 1.0 --lin-d 2 --layernorm --four-seg --drop-ja-2d 1.0`

# Retraining of `RING` -- Older Version

After the installation steps, you can use the two files `train_*.py` to
1) Create training data using `train_step1_generateData.py`. Use `python train_step1_generateData.py --help` for documentation.
    Example: `python train_step1_generateData.py train_xmls/lam1.xml 16`

    For retraining of RING run:
    - file1 (39Gb): `python train_step1_generateData.py train_xmls/lam1.xml 131072 "[standard, expSlow, expFast, hinUndHer]" --seed 1 --sampling-rates "[40, 60, 80, 100, 120, 140, 160, 180, 200]"`
    - file2 (77Gb): `python train_step1_generateData.py train_xmls/lam2.xml 131072 "[standard, expSlow, expFast, hinUndHer]" --anchors "[seg3_2Seg, seg4_2Seg]" --imu-motion-artifacts --seed 2 --sampling-rates "[40, 60, 80, 100, 120, 140, 160, 180, 200]"`
    - file3 (115Gb): `python train_step1_generateData.py train_xmls/lam3.xml 131072 "[standard, expSlow, expFast, hinUndHer]" --anchors "[seg3_3Seg, seg5_3Seg]" --imu-motion-artifacts --seed 3 --sampling-rates "[40, 60, 80, 100, 120, 140, 160, 180, 200]"`
    - file4 (153Gb): `python train_step1_generateData.py train_xmls/lam4.xml 131072 "[standard, expSlow, expFast, hinUndHer]" --anchors "[seg2_4Seg, seg3_4Seg, seg4_4Seg, seg5_4Seg]" --imu-motion-artifacts --seed 4 --sampling-rates "[40, 60, 80, 100, 120, 140, 160, 180, 200]"`

2) (Re)Train RING using `train_step2_trainRing.py`. Use `python train_step2_trainRing.py --help` for documentation.

    For retraining of RING: `python train_step2_trainRing.py path_file1 path_file2 path_file3 path_file4 512 4800 ~/params/trained_ring_params.pickle`
