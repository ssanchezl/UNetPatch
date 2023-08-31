import sys
if __name__ == "__main__":  
	param_stdin = sys.argv[1:]
	params = dict()
	for s in param_stdin:
		key, value = s.split(':')
		params[key] = value


	print(params)

