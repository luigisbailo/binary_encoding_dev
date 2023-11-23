#!/bin/bash


results_dir="./results"

if [ -d "$results_dir" ]; then
	rm -r "$results_dir"
fi
mkdir "$results_dir"


output_dir="./jobs_output"

if [ -d "$output_dir" ]; then
	rm -r "$output_dir"
fi
mkdir "$output_dir"


if [ -e "run_job.sh" ]; then
	rm "run_job.sh"
fi
if [ -e "run_job_bin_enc.sh" ]; then
        rm "run_job_bin_enc.sh"
fi
if [ -e "run_job_no_pen.sh" ]; then
        rm "run_job_no_pen.sh"
fi
if [ -e "run_job_lin_pen.sh" ]; then
        rm "run_job_lin_pen.sh"
fi
if [ -e "run_job_nonlin_pen.sh" ]; then
        rm "run_job_nonlin_pen.sh"
fi

touch run_job.sh

echo '#!/bin/bash -l' >> run_job.sh
echo '#SBATCH --gres=gpu:a100:1' >> run_job.sh
echo '#SBATCH --time=20:00:00' >> run_job.sh
echo '#SBATCH --export=NONE' >> run_job.sh
echo 'unset SLURM_EXPORT_ENV' >> run_job.sh
echo 'module load python' >> run_job.sh
echo 'conda activate bin_enc' >> run_job.sh

sed '2a\#SBATCH --output=./jobs_output/bin_enc%j.out' run_job.sh > ./run_job_bin_enc.sh
sed -i '2a\#SBATCH --job-name=job_bin_enc' run_job_bin_enc.sh

sed '2a\#SBATCH --output=./jobs_output/no_pen%j.out' run_job.sh > ./run_job_no_pen.sh
sed -i '2a\#SBATCH --job-name=job_no_pen' run_job_no_pen.sh

sed '2a\#SBATCH --output=./jobs_output/lin_pen%j.out' run_job.sh > ./run_job_lin_pen.sh
sed -i '2a\#SBATCH --job-name=job_lin_pen' run_job_lin_pen.sh

sed '2a\#SBATCH --output=./jobs_output/nonlin_pen%j.out' run_job.sh > ./run_job_nonlin_pen.sh
sed -i '2a\#SBATCH --job-name=job_nonlin_pen' run_job_nonlin_pen.sh


echo 'python ~/main_training.py --config ./config.yml --model bin_enc --lr 0.0003 --etf-metrics True  --results-dir $2 --sample $1' >> run_job_bin_enc.sh
echo 'python ~/main_training.py --config ./config.yml --model bin_enc --lr 0.0001 --etf-metrics True  --results-dir $2 --sample $1' >> run_job_no_pen.sh
echo 'python ~/main_training.py --config ./config.yml --model bin_enc --lr 0.0001 --etf-metrics True  --results-dir $2 --sample $1' >> run_job_lin_pen.sh
echo 'python ~/main_training.py --config ./config.yml --model bin_enc --lr 0.0001 --etf-metrics True  --results-dir $2 --sample $1' >> run_job_nonlin_pen.sh



for i in {1..3}
do 
	sbatch ./run_job_bin_enc.sh $i $results_dir
   	sbatch ./run_job_no_pen.sh $i $results_dir
   	sbatch ./run_job_lin_pen.sh $i $results_dir
   	sbatch ./run_job_nonlin_pen.sh $i $results_dir

done


cp ./config.yml "$results_dir"/config.yml

rm run_job*
