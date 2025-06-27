import crossfiledialog
import os

with open("config.toml", "r") as infile:
    with open("config-new.toml", "w") as outfile:
        for line in infile:
            # Input images
            if line.startswith("input_images"):
                input_images = crossfiledialog.choose_folder(title="Choose the directory where the camera folders are stored.").replace("\\", "/")
                outfile.write(f"input_images = \"{input_images}\"\n")
            # Images output
            elif line.startswith("images_output"):
                images_output = crossfiledialog.choose_folder(title="Choose the directory where processed images should go.").replace("\\", "/")
                outfile.write(f"images_output = \"{images_output}\"\n")
            # Trainee model
            elif line.startswith("trainee_model"):
                trainee_model = crossfiledialog.choose_folder(title="Choose the directory where the downloaded model should be saved.").replace("\\", "/")
                outfile.write(f"trainee_model = \"{trainee_model}/trainee_model.pth\"\n")
            # Trained model and model output
            elif line.startswith("trained_model"):
                models_output = crossfiledialog.choose_folder(title="Choose the directory where trained models should go.").replace("\\", "/")
                outfile.write(f"trained_model = \"{models_output}/model.pth\"\n")
                outfile.write(f"models_output = \"{models_output}\"\n")
            elif line.startswith("models_output"):
                pass
            else:
                outfile.write(line)

os.remove("config.toml")
os.rename("config-new.toml", "config.toml")