#!/bin/bash
# Evaluate the MACE-LES total potential energy of every dumped frame of one state.
SID=$1
D=/publicfs10/fs10-m9/home/m9s005258/qy/cdft/scan/traj/$SID/run
OUT=/publicfs10/fs10-m9/home/m9s005258/qy/cdft/energy_out
CSV=$OUT/$SID.energy.csv
[ -s "$CSV" ] && [ "$(wc -l < $CSV)" -ge 2001 ] && exit 0
TMP=${SLURM_TMPDIR:-/tmp}/ev_$SID
mkdir -p "$TMP"
zcat $D/traj.dump.gz > $TMP/traj.dump
PS=$(grep "^pair_style" $D/in.les_md); PC=$(grep "^pair_coeff" $D/in.les_md)
DATA=$(grep "^read_data" $D/in.les_md | awk '{print $2}')
cat > $TMP/in.e <<EOT
units metal
atom_style molecular/kk
atom_modify map yes
boundary p p p
read_data $DATA
$PS
$PC
neighbor 2.0 bin
neigh_modify every 10 delay 0 check yes
run_style verlet/kk
thermo 1
thermo_style custom step atoms pe press
rerun $TMP/traj.dump dump x y z
EOT
/publicfs10/fs10-m9/home/m9s005258/qy/build-cpu-mpi/lmp -k on t 1 -in $TMP/in.e -log $TMP/run.log > /dev/null 2>&1
awk '/^ *[0-9]+ +[0-9]+ +-?[0-9.]+/ && $2>0 {print $1","$3","$4}' $TMP/run.log | awk '!seen[$1]++' > "$CSV"
echo "$SID $(wc -l < $CSV) $(date +%H:%M:%S) $(hostname)" >> $OUT/progress.txt
rm -rf "$TMP"
