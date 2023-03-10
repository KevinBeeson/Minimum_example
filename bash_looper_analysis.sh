OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 
MKL_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
NUMEXPR_NUM_THREADS=1
input="minimal_run.txt"
while IFS= read -r line
do
  #echo $line"_prior.txt" > run_out_puts/$line"_prior.txt" &
  python3 -u main_analysis.py -s $line -n 1  &> run_out_puts/$line".txt" &
done < "$input"
