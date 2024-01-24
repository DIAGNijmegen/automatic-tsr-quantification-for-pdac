#sudo -H -E -u user jupyter notebook --ip=0.0.0.0 --port=8888 --NotebookApp.token=
#!/bin/bash


# Check if extra arguments were given and execute it as a command.
#
if [ -z "$2" ]
    then
        # Print the command for logging.
        
        
 #        cd /home/user
        
 #        # install smooth topk for clam
 #        git clone https://github.com/oval-group/smooth-topk.git
 #        cd smooth-topk
        
 #        echo 'user' | sudo -S python3.8 setup.py install
 #        pip3.8 install jenkspy 
 #        pip3.8 install histomicstk --find-links https://girder.github.io/large_image_wheels
	# cp -r /data/pathology/projects/pathology-bigpicture-dlbcl-myc/pathology-encoders /home/user/pathology-encoders
 #        echo "No extra arguments given, running jupyter and sshd"
 #        echo
 #        echo "PYTHONPATH IS"
	# export PYTHONPATH=$PYTHONPATH:/home/user/pathology-encoders
	# echo $PYTHONPATH
 #        # Start the SSH daemon and a Jupyter notebook.
        
        
        /usr/sbin/sshd
    export PYTHONPATH="${PYTHONPATH}:/opt/ASAP/bin:/home/user/source/pathology-common:/home/user/source/pathology-fast-inference"
	cd /home/user && sudo  --set-home --preserve-env --user=user  /bin/bash -c '/usr/local/bin/jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='
	#sleep 86400

    else
        # Print the command for logging.
        #
        echo "Execute command: ${@}"
        echo

        # Execute the passed command.
        # python setup.py install
        cd /home/user && sudo --user=user --set-home "${@}"
fi

