REL_PATH=../../
DIR_NAME="${PWD##*/}"
EXP_DIR="exp-$(date "+%Y-%m%d-%H%M%S")"

#python "${REL_PATH}seatable.py" "${EXP_DIR}" &

PYTHONPATH=${PYTHONPATH}:${REL_PATH} GLOG_vmodule=MemcachedClient=-1 \
spring.submit run -r --gpu -n$1 \
--ntasks-per-node=$1 \
--cpus-per-task=5 \
--job-name "${DIR_NAME}" \
"python -u -m main --main_py_rel_path=${REL_PATH} --exp_dirname=${EXP_DIR} \
--batch-size 128 \
--epochs 501 \
--lr 5e-4 \
--min-lr 5e-6 \
--training-mode 'finetune' \
--data-set 'stl10' \
--finetune '/expxxxxx/checkpoints/SSL/STL10/checkpoint.pth'"

failed=$?
echo "failed=${failed}"

RESULT=$(tail "${EXP_DIR}"/log.txt -n 1)
echo ""
echo -e "\033[36mat ${PWD#}/${EXP_DIR}\033[0m"
echo -e "\033[36m${RESULT#*@}\033[0m"

#if [ $failed -ne 0 ]; then
#  sh "./kill.sh"
#  echo "killed."
#else
#  touch "${EXP_DIR}".terminate
#fi


