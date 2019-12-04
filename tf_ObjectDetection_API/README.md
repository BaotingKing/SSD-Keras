使用方法:
1. 修改对应模型的*.config文件,需要设置dataset\Finetune\一些超参数.
2. 训练结束后,需要将模型进行转换,生成对应的.pb文件,利用object_detection文件下export_inference_graph函数,命令行格式如下:
	python3 export_inference_graph.py 
		--input_type image_tensor 
		--pipeline_config_path proc_train/ssd_mobilenet_v2_coco.config 
		--trained_checkpoint_prefix /eDisk/Zack_SSD_mv2/model.ckpt-445889 
		--output_directory /eDisk/Zack_SSD_mv2/005_ssdmv2
	最后模型生成在--output_directory指定的路径下,文件名为frozen_inference_graph.pb
3. 执行object detection程序进行mAP统计, 需要用到
