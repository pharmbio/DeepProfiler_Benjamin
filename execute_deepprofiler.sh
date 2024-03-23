#source ~/.bashrc

#export DBI_URI=postgresql://pharmbio_readonly:readonly@imagedb-pg-postgresql.services.svc.cluster.local/imagedb

#echo $DBI_URI

python deepprofiler_dataprep.py --projectfolder /home/jovyan/share/data/analyses/benjamin/Single_cell_project/Deep_test --metadata /home/jovyan/share/data/analyses/benjamin/Single_cell_project/AROS-Reproducibility-MoA-Full.csv --projectname AROS --mode metadata 
