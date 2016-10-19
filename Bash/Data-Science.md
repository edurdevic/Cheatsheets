Useful commands to manipulate data at the command line

# Docker image with essential tools for Data Science, by Jeroen Janssens
## Pull the Docker image
 docker pull jeroenjanssens/data-science-at-the-command-line
## Run the Docker container
 docker run -it -v $(pwd):/data jeroenjanssens/data-science-at-the-command-line bash
                      ^ mounts the current dir into the container at /data 


# JQ: Processing Json inputs
 < asd.json jq .                                 # Prints nicely a json
 seq 10000 | sample -d 10000 | jq '{value: .}'
 < table.json jq '.' -C | less -R                # Shows the currrent object ('.') and colour the output
 < table.json jq '.html.body' -C
 < table.json jq '.html.body.tr[]' -C            # Select elements on '.html.body.tr[]'
    # format the selection '.html.body.tr[]' in a new object '{ country: value }', where value is .td[1][]
 < table.json jq '.html.body.tr[] | { country: .td[1][] }' -C | head -n 10     

## JQ FILTER: 
  # Filter bikes json by station "Clinton St". 
   < citybikes.json jq -r '.[] | [ .name, .bikes] | @csv' | grep "Clinton St"
   < citibikes.json jq -r '.[] | select(.name | contains("1 Pl & Clinton St")) | .bikes'

# SCRAPE: tool in python that uses beautifulsoup to scrape html
 < wiki.html scrape -b -e 'table.wikitable > tr:not(:first-child)' > table.html    

# XML2JSON
 < table.html xml2json > table.json

# JSON2CSV
 < coutries.json json2csv > countries.csv
 < coutries.json json2csv -L > countries.csv          # If the json in line delimited json objects instead of a valid JSON array

# CSVLOOK: Formats csv file as a table
 csvlook countries.csv

# CSVSQL: Query csv as SQL using SqlLite
 < countries.csv csvsql --query 'select * from stdin order by ratio desc limit 5' 
 csvsql --query 'select * from countries order by ratio desc limit 5' countries.csv 

# HEADER: Adds a header line to a file
 seq 10 | header -a value | csvsql --query 'select sum(value) from stdin'
 
# GNU PARALLEL: Runs a sequence of tasks
 parallel -h 
 seq 10 | parallel "cowsay {}"        # executes cowsay 10 times with numbers from 0 to 10
 seq 10 | parallel "touch file{}.csv"
 find -name '*.csv' | parallel -t "mv {} {.}.json"
 parallel --jobs 1 --dry-run    --delay 0.1 --results results "curl -sL 'http://api.nytimes.com/svc/search/v2/articlesearch.json?q=New+York+Fashion+Week&begin_date={1}0101&end_date={1}1231&page={2}&api-key=ad8185530d4853d52db1f9900738cda5:11:68109136'" ::: {2009..2011} ::: {0..2}
                    ^ "What if": shows what will happen when it Runs
 parallel --jobs 1 --progress   --delay 0.1 --results results "curl -sL 'http://api.nytimes.com/svc/search/v2/articlesearch.json?q=New+York+Fashion+Week&begin_date={1}0101&end_date={1}1231&page={2}&api-key=ad8185530d4853d52db1f9900738cda5:11:68109136'" ::: {2009..2011} ::: {0..2} > /dev/null
 parallel --jobs 1 --bar        --delay 0.1 --results results "curl -sL 'http://api.nytimes.com/svc/search/v2/articlesearch.json?q=New+York+Fashion+Week&begin_date={1}0101&end_date={1}1231&page={2}&api-key=ad8185530d4853d52db1f9900738cda5:11:68109136'" ::: {2009..2011} ::: {0..20} > /dev/null

# SED: Extract and manipulate text
 cat x.txt | sed -n '2p'                            # get the second line of a text
 cat x.txt | sed 's/ /_/g'                          # replace space with underscore
 cat x.txt | sed -n '/^Subject://;p'                # get only the line which starts with "Subject:". "p" stands for "print" and -n for no output
 cat x.txt | sed -n '/^Subject:/{s/Subject: //;p}'  # same as above, but substitute ""

# TR: Translate
 echo "HELLO" | tr '[A-Z]' '[a-z]'
 echo "HELLO" | tr '[:upper:]' '[:lower:]'

