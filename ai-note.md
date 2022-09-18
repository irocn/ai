## Environment setup 

1. python upgrade 3.9  
$ apt install python3.9  
$ sudo update-alternatives --config python3 (select python3.9)  

If you are not planning to use the old version of Python, remove the symlink that contained the previous Python 3 version with:  
sudo rm /usr/bin/python3  
Then, replace the symlink with the new version:  
sudo ln -s python3.9 /usr/bin/python3  
Now, check the default version:  
python3 --version  
