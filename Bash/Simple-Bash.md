

# Change prompt (you can configure it to show pwd, git branch, etc.):
 PS1="$ "
 PS1="\[$(tput sgr0)\]\[\033[38;5;6m\][\w]$\[$(tput sgr0)\]\[\033[38;5;15m\] \[$(tput sgr0)\]"

# CAT: concatenate multiple files
 cat file1.txt file2.txt

# SEQ: Sequence
 seq 10        # sequenza da 0 a 10
 seq 1 10      # da 1 a 10
 seq 1 2 10    # ogni secondo numero

# WC: Word count
 wc -l         # Number of lines

# LESS: paginator
less -S asd        # -s = do not word wrap

# SORT: sort
 sort -r       # Sort reverse 
 sort -nr      # Sort number reverse

# Passing files as input to commands:
 cat numbers | grep 3 | head     # Read from "numbers" file and pass to grep, show first 10 lines
 < numbers grep 3 | head         # Read from "numbers" file and pass to grep, show first 10 lines
 grep 3 < numbers | head         # Read from "numbers" file and pass to grep, show first 10 lines
 grep 3 numbers | head           # Read from "numbers" file and pass to grep, show first 10 lines

# Combine moultiple tools into a pipeline
 mkdir && cd         # Execute mkdir and than cd only if the first was successful
 mkdir ; cd          # Execute mkdir and cd anyway

# Pass env vars to a command:
 MY_VAR=10 command


# ECHO: writes arguments to the standard output
 echo results/1/*/2/*/stdout    # 
 echo {2005..2015}              # Result: 2005 2006 2007 ... 2015
 echo {2005..2015}
 echo "$(date)" | cowsay        # Evaluate a command as parameter using duble qoutes
 echo '$(date)' | cowsay        # Literal, not evaluated
 < countries.csv csvsql --query "$(cat top5.sql)" | csvlook

# CURL: transfer  data from or to a server
 curl -I http://bit.ly/iris-csv     # Get document info only
 curl -L http://bit.ly/iris-csv     # Get the resurce by following redirects 301
 curl -sL 'http://en.wikipedia.org/wiki/List_of_countries_and_territories_by_border/area_ratio'  

# CUT: cuts out selected portions of each line
 < file.txt cut -c 1-70    # Show only the first 70 characters per line

# GREP:
 < wiki.html grep wikitable -A 21            # Find the line with "wikitable" and 21 lines after
 < wiki.html grep wikitable -C 1             # Find the line with "wikitable" and 1 before and after

# TREE: Tree representation of the current dir
 tree 

# Create a CUSTOM bash function:
 my_function () {     
    tr '[:upper:]' '[:lower:]' ;
 }

# Create CUSTOM python tools:
 1) Create a file with shebang: "#!/usr/bin/python2"
 2) Give execution permission to the created file: chmod +x filename
 3) add your script directory to PATH 