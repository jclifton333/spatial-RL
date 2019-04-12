# Install git
sudo apt-get update
sudo apt-get install git --user

# Clone repo 
git clone https://github.com/jclifton333/spatial-RL


# Install pip
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py

# Install dependencies
cd spatial-RL
git pull --all
git checkout -b gravity
python -m pip install -r requirements.txt --user


