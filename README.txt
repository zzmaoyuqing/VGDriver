
1.Firstly, run dataset/generate_dataset.py to build dataset.pkl for each PPI.
2.Then, run main.py to reproduce.
Run the command line in a terminal 'python main.py --cacer_type [pancancer or BRCA or LUAD or PRAD or KIRC or LIHC or LUSC or STAD or THCA]' --network [BioGRID or CPDB or HINT or IRefIndex or STRING]

Notice： If you want to use other cancer types which are not contained in the 16 cancer types, you need to refer the 'data_preprocessing' folder to process omics data or PPI data steps by steps.
Our method covers 16 cancer types and 5 PPI networks (BioGRID, CPDB, HINT, IRefIndex and STRING).
On pan-cancer we compared with existing methods in 5 PPI networks. Specially, we also compared with existing methods on eight single cancers (BRCA, LUAD, PRAD, KIRC, LIHC, LUSC, STAD, and THCA) in five PPI networks as well.

VGDriver requires the following dependencies:
python==3.7.0
torch==1.10.0
torchvision==0.11.0
torchaudio==0.10.0
cudatoolkit==11.3
torch_cluster-1.6.0
torch_sparse-0.6.12
torch_scatter-2.0.9
torch-geometric==2.0.3
pandas==1.1.5
scikit-learn==1.0.2