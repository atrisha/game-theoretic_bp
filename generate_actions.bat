@echo off
IF [%1] == [] (ECHO Please enter a file id from one of 769 770 771 772 775 776 777 778 779 780 781 782 783 784 785) ELSE GOTO ok

:ok
ECHO Generation of actions started....
python .\runner_trajectory_generation.py %1 BOUNDARY && python .\runner_trajectory_generation.py %1 BASELINE && python .\runner_trajectory_generation.py %1 GAUSSIAN

