MAIN = GCG_test

include ./make.inc
# If you want to check the momory, please use 
# valgrind --leak-check=full ./main > check

$(MAIN) : $(OBJECTS)
	$(CC) $(CPPFLAGS)  $(CFLAGS)  -o $@ $^ $(LIBS)
.c.o:
	$(CC) $(CPPFLAGS)  $(CFLAGS)  -o $*.o -c $*.c

	
clean : 
	rm -f $(OBJECTS) 
	rm -rf $(MAIN)
	rm -rf *~
