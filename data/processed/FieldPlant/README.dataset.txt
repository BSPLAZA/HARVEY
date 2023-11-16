# FieldPlant > fieldplant_dataset
https://universe.roboflow.com/plant-disease-detection/fieldplant

Provided by a Roboflow user
License: CC BY 4.0

**Overview**

The Food and Agriculture Organization of the United Nations suggests increasing the food supply by 70% to feed the world population by 2050, although approximately one third of all food is wasted because of plant diseases or disorders. To achieve this goal, researchers have proposed many deep learning models to help farmers detect diseases in their crops as efficiently as possible to avoid yield declines. These models are usually trained on personal or public plant disease datasets such as PlantVillage or PlantDoc. PlantVillage is composed of laboratory images captured under laboratory conditions, with one leaf each and a uniform background. The models trained on this dataset have very low accuracies when running on field images with complex backgrounds and multiple leaves per image. To solve this problem, PlantDoc was built using 2,569 field images downloaded from the Internet and annotated to identify the individual leaves. However, this dataset includes some laboratory images and the absence of plant pathologists during the annotation process may have resulted in misclassification. In this study, FieldPlant is suggested as a dataset that includes 5,170 plant disease images collected directly from plantations. Manual annotation of individual leaves on each image was performed under the supervision of plant pathologists to ensure process quality. This resulted in 8,629 individual annotated leaves across the 27 disease classes. We ran various benchmarks on this dataset to evaluate state-of-the-art classification and object detection models and found that classification tasks on FieldPlant outperformed those on PlantDoc.

Cite this research:

@ARTICLE{10086516,
  author={Moupojou, Emmanuel and Tagne, Appolinaire and Retraint, Florent and Tadonkemwa, Anicet and Wilfried, Dongmo and Tapamo, Hyppolite and Nkenlifack., Marcellin},
  journal={IEEE Access}, 
  title={FieldPlant: A dataset of field plant images for plant disease detection and classification with deep learning}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/ACCESS.2023.3263042}}