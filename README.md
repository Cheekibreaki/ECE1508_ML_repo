# Lightweight Diffusion Architecture for Fashion Text-to-Image Generation

## Group Members:
Jacqueline Zhu: jingqi.zhu@mail.utoronto.ca

Yihang Lin: yihang.lin@mail.utoronto.ca

Ji Ping Li: jiping.li@mail.utoronto.ca

Ziang Jia: ziang.jia@mail.utoronto.ca

## Project Structure
    ECE1508-ML_repo/
        ├── data_sampling/
        │   └── main.py  # script for data generation pipline 
        │   └── trim.py  # script for trimming data label within any given specific lenth (For experiment run only)
        │   └── data_9067.zip  # 9067 pieces of data used for model training + testing
        ├── final_model/
        │   └── presentation_UNet_V2.ipynb # final ipynb file deliverable
        |   └── model_demo_and_eval.ipynb  # Same model as above, except model is initialized through loading pth instead of training. Then computes the FID and Clip score. 
        |   └── pooled_9067_full_checkpoint.pth # pth for model parameter loading
        ├── past_models/
        │   ├── UNet_V2_with_dynamic_thresholding.ipynb
        │   ├── dynamic_thresholding_cfg_unet_v2_with_cross_attention_model.pth
        │   ├── simple_diffusion_BERT.ipynb    # First Diffusion model prototype
        │   └── UNet_with_cross_attention.ipynb    # First UNet version
        ├── results/
        │   ├── final_results # directory containing final results
        │   └── past_results # directory containing results from past models
        ├── requirements.txt
        └── README.md

## Set up Instruction
1. Go to repository home directory and install pacakges using the following command: 

        pip install -r requirements.txt


2. It is recommended to run the .ipynb files in Google Colab. 
        Move presentation_UNet_V2_perf_demo.ipynb into colab,
        Move data_9067.zip and dynamic_thresholding_cfg_unet_v2_with_cross_attention_model.pth into local google myDrive space
4. To use specific zipped dataset in Google Colab, please mount drive and unzip the dataset files: 

        from google.colab import drive
        drive.mount('/content/drive')

        !unzip -q "/content/drive/MyDrive/<..... YOUR PATH .....>/data_full_9067.zip" -d /content/dataset  

        ckpt_dir  = "/content/drive/MyDrive/<..... YOUR PATH .....>"                
        ckpt_name = "pooled_9067_full_checkpoint.pth"
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)


