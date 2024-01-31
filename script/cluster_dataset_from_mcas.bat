(
    cd ..\src\mca
    conda activate dm
    conda run -n dm --no-capture-output --live-stream python cluster_dataset_from_mcas.py --cluster_size=3 --cluster_stride=3 --parallelize --max_concurrent_processes=20
    conda deactivate
    cd ..\..\script
)