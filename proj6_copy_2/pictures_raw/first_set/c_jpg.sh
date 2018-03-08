for i in *.CR2; do sips -s format jpeg $i --out "${i%.*}.jpg"; done
