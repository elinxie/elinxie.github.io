from os import listdir
from os.path import isfile, join
import sys

def basic_site_body_wrapper(my_str):
    site = ""
    site += "<!doctype html>\n\t<html>\n\t\t<head>\n\t\t\t<title>picture website</title>\n\t\t</head>\n\t<body>"
    site += my_str
    site += "\n\t</body>\n</html>"
    return site

def image_plus_disc(lst1, lst2) :
	# always display all of the pictures, do not have more descriptions than pictures
	if (len(lst1) < len(lst2)):
		lst2 = lst2[:len(lst1)]
	if (len(lst1) > len(lst2)):
		lst2 += [''] * (len(lst1) - len(lst2))
	body_html_lst = ["\n\t<div>\n\t\t<img src='{0}' width='640px' >\n\t\t<p>\n\t\t\t{1}\n\t\t</p>\n\t</div>".format(a,b)
						for a, b in zip(lst1,lst2)]
	return "".join(body_html_lst)

def get_pics_from_dir(my_path):
    return [join(my_path, f) for f in listdir(my_path) if isfile(join(my_path, f))]

def get_descriptions(my_file_str):
	my_file = open(my_file_str)
	my_descriptions = my_file.readlines()
	my_file.close()
	return my_descriptions

def is_full_site(my_flag):
	return my_flag == 'full-site'

def correct_flag(argv):
	my_flag = argv[4]
	return is_full_site(my_flag) and len(argv) == 5

def check_for_DS_store(my_pics):
	for i in range(0, len(my_pics)):
		if my_pics[0].split(".")[-1] == "DS_Store":
			del my_pics[0]
	return my_pics

def main(argv):
	# we have to make sure that we have the correct flags, not put in junk. 
	if (len(argv) == 4 or correct_flag(argv)):
		pics_path = get_pics_from_dir(argv[1])
		my_descriptions = get_descriptions(argv[2])
		pics_path = check_for_DS_store(pics_path)
		site = image_plus_disc(pics_path, my_descriptions)
		if (correct_flag(argv)):
			site = basic_site_body_wrapper(site)
		webpage = open(argv[3],"w+")
		webpage.write(site)
		webpage.close()
	else:
		print("""
			to run, use 
			python pic_site.py /pathto/pictures descriptions.txt output.html full-site 
			where 
			- 'full-site' is a tag used to just give html code of not only the pictures but of wrapping html code
			so that the resulting text can be made into a stand-alone html page. By default, the script will
			get the html string to embed into your own html page. 
			- '/pathto/pictures' is the directory to the pictures you want to list in an html page. 
			Make sure the pictures are not in subdirectories.
			- 'descriptions.txt' is a textfile that contains all the discriptions you will put under the pictures. 
			Each description is seperated by a new line
			- 'output.html' is the textfile where we will put the resulting html code
			""")

if __name__ == "__main__":
    main(sys.argv)
