python models.py --single_step 3 --window_size 20 --rootdir ../Logs --label_dir ../labels/D2020/ --vector_path ../Logs/template_vec.dat --selected_switchid_list [14,15,16,17,18]

python transferlearning.py --single_step 3 --window_size 20 --rootdir ../Logs --label_dir ../labels/B6220/ --vector_path ../Logs/template_vec.dat  --selected_switchid_list [0, 1, 2, 3]

python predict.py --single_step 3 --window_size 20 --rootdir ../Logs --label_dir ../labels/B6220/ --vector_path ../Logs/template_vec.dat  --selected_switchid_list [11] --save_model_dir ../checkpoint/transfer/ --save_result_dir ../result/ --mode predict

python predict.py --single_step 3 --window_size 20 --rootdir ../Logs --label_dir ../labels/B6220/ --vector_path ../Logs/template_vec.dat --selected_switchid_list [11] --save_model_dir ../checkpoint/transfer/ --save_result_dir ../result/ --mode performance
