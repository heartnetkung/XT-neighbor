# for first time run these
# mkdir gh-pages
# cd gh-pages
# git clone git@github.com:heartnetkung/XT-neighbor.git

doxygen Doxyfile
cd gh-pages/XT-neighbor
git add .
git commit -m "update doc"
git push
