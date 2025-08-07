# Lightweight Diffusion Architecture for Fashion Text-to-Image Generation

## Group Members:
Jacqueline Zhu: jingqi.zhu@mail.utoronto.ca

Yihang Lin: yihang.lin@mail.utoronto.ca

Ji Ping Li: jiping.li@mail.utoronto.ca

Ziang Jia: ziang.jia@mail.utoronto.ca

## Project Structure
    ECE1508-ML_repo/
        ├── data_sampling/
        │   ├── data_generation_0-99_2025-07-02   # dataset 100 samples
        │   └── main.py  # script for data generation pipline 
        ├── final_model/
        │   └── presentation_UNet_V2.ipynb # final ipynb file deliverable
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

3. To use specific zipped dataset in Google Colab, please mount drive and unzip the dataset files: 

        from google.colab import drive
        drive.mount('/content/drive')

        !unzip -q "/content/drive/MyDrive/<..... YOUR PATH .....>/data_full_9067.zip" -d /content/dataset  
    


