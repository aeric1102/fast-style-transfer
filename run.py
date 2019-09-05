import subprocess
import os

filenames = os.listdir("./data/style")

for filename in filenames:
    style = os.path.splitext(filename)[0]
    cmd = ("python style.py --style ./data/style/" + filename +
     " --checkpoint-dir ./data/sm_models/" + style + 
     " --test ./data/sm_test/test.jpg" +
     " --test-dir ./data/sm_test/" + style +
     " --batch-size 4" + 
     " --gpu-fraction 0.3")
    result = subprocess.check_output(cmd, shell=True)


    #     ["python", "style.py", 
    #      "--style", "./data/style/" + filename,
    #      "--checkpoint-dir", "./data/sm_models/" + style,
    #      "--test", "./data/sm_test/test.jpg",
    #      "--test-dir", "./data/sm_test/" + style,
    #      "--batch-size", "4",
    #      "--gpu-fraction", "0.3"]
    #      , shell=True)