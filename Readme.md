## CCBDA HW3 - Diffusion Generative Model
(Dataset MNIST) 60000 handwritten digits in size 28x28.
- Implementing difussion generative models.
- Evaluating generative models in terms of FID.

## Dataset
- Unzip data.zip to `./`
    ```sh
    unzip data.zip 
    ```
- Folder structure
    ```
    ./
    ├── mnist/ 
    ├── main.py   
    ├── mnist.npz 
    ├── requirements.txt
    ├── Unet.py
    └── utils.py
    ```

## Environment
- Python 3.6 or later version
    ```sh
    pip install -r requirements.txt
    ```

## Train
```sh
python main.py
```
- Generated 10000 Images
The generated images folder is `images`

- Record the Diffusion Process
The diffusion process file is `process_result.png`
