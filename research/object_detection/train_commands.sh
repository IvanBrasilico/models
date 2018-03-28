. padma-venv/bin/activate
cd models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

python object_detection/dataset_tools/create_pascal_tf_record.py --label_map_path=object_detection/classes.pbtxt --data_dir=/home/ivan/pybr/conteineres_rotulados/ --set=train --output_path=object_detection/treinamento/coco_train.record

# Inception
python object_detection/train.py --logtostderr --pipeline_config_path=object_detection/treinamento/ssd_inception_v2_coco.config --train_dir=object_detection/treinamento/models/ssd_inception/trained
python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path object_detection/treinamento/ssd_inception_v2_coco.config --output_directory object_detection/mygraph --trained_checkpoint_prefix object_detection/treinamento/models/ssd_inception/trained/model.ckpt-

# Mobilenet
python object_detection/train.py --logtostderr --pipeline_config_path=object_detection/ssd.config --train_dir=object_detection/treinamento/models/ssd_mobilenet/trained
python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path object_detection/ssd.config --output_directory object_detection/mygraph --trained_checkpoint_prefix object_detection/treinamento/models/ssd_mobilenet/pretrained/model.ckpt-
