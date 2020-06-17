for i in {1..1};
do
 python cmotp.py --environment CMOTP_V1 --processor '/gpu:0' --madrl hysteretic
done
