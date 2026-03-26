for file in dir1/*.txt; do diff -q "$file" "dir2/${file##*/}"; done
