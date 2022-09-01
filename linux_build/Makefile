DIR := ../coupling_speed_test
EXT_LIBS := ../coupling_speed_test/external_libs
CC := nvcc
EXTRA := # Specify extra flags when running make with make EXTRA=[flags...]
NVCC := nvcc

# ############################################################################################################
# ### SPECIFY LIBRARIES HERE. THIS IS FOR A NON-OPENGL BUILD. ################################################
# ############################################################################################################
LINUX_LIBS := $(EXT_LIBS)/ann_1.1.2 $(EXT_LIBS)/glm $(EXT_LIBS)/hdf5-1.10.8 $(EXT_LIBS)/json

# Setup command to handle all the library includes.
INCLUDE := $(foreach lib, $(LINUX_LIBS), $(lib)/include)
HANDLE_INCLUDES = $(foreach include_path, $(INCLUDE), -I$(include_path))

# Setup command to handle all the library linking.
LIB := $(foreach lib, $(LINUX_LIBS), $(lib)/lib/) 
HANDLE_LIB_DIRS := $(foreach dir, $(LIB), -L$(dir))

# Get .a files and remove path, lib prefix, and .a suffix.
LIB := $(foreach lib, $(LIB), $(wildcard $(lib)/*.a))
LIB := $(foreach lib, $(LIB), $(notdir $(lib))) # Remove path from libs
LIB := $(LIB:lib%.a=%) # Remove lib prefix and .a suffix from files
HANDLE_LIB_FILES := $(foreach lib, $(LIB), -l$(lib))

# ############################################################################################################
# ### SOME EXTRA LIBRARIES ARE OFTEN NEEDED. THEY ARE ADDDED HERE. ###########################################
# ############################################################################################################
HANDLE_LIB_FILES += -lz -ldl -lm # Extra things needed for hdf5

HANDLE_LIBS := $(HANDLE_LIB_DIRS) $(HANDLE_LIB_FILES)

# Flags. 
CFLAGS := $(HANDLE_INCLUDES) $(EXTRA)
NVFLAGS := $(HANDLE_INCLUDES) $(EXTRA)
TARGET := coupling_speed_test

CPPFILES := $(wildcard $(DIR)/*.cpp)
CUFILES := $(wildcard $(DIR)/*.cu)

HEADERS = $(DIR)/*.h
OBJECTS = $(foreach file, $(CPPFILES) $(CUFILES), $(notdir $(basename $(file))).o) # This line replaces the cpp and cu paths with .o names

main: main.o $(OBJECTS)
	$(NVCC) $(OBJECTS) $(HANDLE_LIBS) -o $@S $(NVFLAGS)

%.o: $(DIR)/%.cu 
	$(NVCC) -c $< -o $@ $(NVFLAGS)

%.o: $(DIR)/%.cpp
	$(CC) -c $< -o $@ $(CFLAGS)

clean:
	-rm -f $(OBJECTS)
	-rm -f main 

debug: 
	@echo
	@echo C++ FILES: $(CPPFILES)
	@echo
	@echo CU FILES: $(CUFILES)
	@echo
	@echo OBJECT FILES TO COMPILE: $(OBJECTS)
	@echo
	@echo COMMAND TO HANDLE INCLUDES: $(HANDLE_INCLUDES)
	@echo
	@echo COMMAND TO HANDLE LIBS: $(HANDLE_LIBS)
