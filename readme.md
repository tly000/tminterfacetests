###Usage

- Generate Dataset:
  - copy the `TmNationsForever\GameData\Tracks\Campaigns\Nations\White` folder to the place where you want to create the dataset
  - start TMInterface.exe in 640x480 windowed mode, log in and close the TMInterface console
  - run `python generate_dataset.py Path/TmNationsForever/TmForever.exe Path/Of/Copied/WhiteCampaign`
  - start any challenge and select the camera that you want for the videos
  - wait for the script to generate `.mp4` and `.states.bin` files for all challenges
    - you might have to click on "Ok" if you gain a medal while generating...
    - the TmForever window should stay in focus, otherwise the framerate of the game is automatically decreased.

- Imitation learning:
  - run `python imitation_learning.py Path/Of/Copied/WhiteCampaign`
  - wait a bit... after every epoch "train.pth" is saved in the folder

- let the model play a track:
  - make sure that no tm window is open
  - run `python run_model.py Path/Of/Copied/WhiteCampaign`
