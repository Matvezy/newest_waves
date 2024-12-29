import torch

# List of file paths
list_of_names = [
    './results_llamagen/real_1_resizedcrop_0.05/results_a_0.05.pt', 
    './results_llamagen/real_1_resizedcrop_0.1/results_a_0.1.pt', 
    './results_llamagen/real_1_resizedcrop_0.2/results_a_0.2.pt', 
]
"""
'./results_llamagen/real/results_a_real.pt', 
    './results_llamagen/none-2-real/results_a_none-2-real.pt', 
    './results_llamagen/resnet18-2-real/results_a_resnet18-2-real.pt', 
    './results_llamagen/resnet18_pwm-2-real/results_a_pwm-2-real.pt',
    './results_llamagen/real_resizedcrop/results_a_resizedcrop.pt', 
    './results_llamagen/none-2-real_resizedcrop/results_a_resizedcrop.pt',
    './results_llamagen/real_resizedcrop_0.05/results_a_0.05.pt', 
    './results_llamagen/real_resizedcrop_0.1/results_a_0.1.pt', 
    './results_llamagen/real_resizedcrop_0.2/results_a_0.2.pt', 
    './results_llamagen/none-2-real_resizedcrop_0.05/results_a_0.05.pt', 
    './results_llamagen/none-2-real_resizedcrop_0.1/results_a_0.1.pt', 
    './results_llamagen/none-2-real_resizedcrop_0.2/results_a_0.2.pt',
"""

# Define function to process scores
def process_scores(file_path, label):
    data = torch.load(file_path)['tensor_list']
    average_score = sum(item['z_score'] for item in data) / len(data)
    number_of_true = sum(1 for item in data if item['prediction'] == 1)
    number_of_false = len(data) - number_of_true
    
    # Print results
    print(f"{label}:")
    print("Average z-score:", average_score)
    print("Number of true predictions:", number_of_true)
    print("Number of false predictions:", number_of_false)
    print("\n")

"""
# Process each file with corresponding label
process_scores(list_of_names[0], "Regular")
process_scores(list_of_names[1], "Simple 1 WM")
process_scores(list_of_names[2], "Regular ResNet")
process_scores(list_of_names[3], "Simple 1 WM ResNet")
process_scores(list_of_names[4], "Cropped Regular")
process_scores(list_of_names[5], "Cropped WM")

process_scores(list_of_names[6], "Regular - 0.05")
process_scores(list_of_names[7], "Regular - 0.1")
process_scores(list_of_names[8], "Regular - 0.2")
process_scores(list_of_names[9], "Simple 1 WM - 0.05")
process_scores(list_of_names[10], "Simple 1 WM - 0.1")
process_scores(list_of_names[11], "Simple 1 WM - 0.2")
"""
process_scores(list_of_names[0], "Simple Fixed 1 WM - 0.05")
process_scores(list_of_names[1], "Simple Fixed 1 WM - 0.1")
process_scores(list_of_names[2], "Simple Fixed 1 WM - 0.2")