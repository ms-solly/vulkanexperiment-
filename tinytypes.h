#ifndef TINYTYPES_H
#define TINYTYPES_H

#include <stdint.h>
#include <stddef.h>  // for size_t
#include <stdbool.h> // for bool

#define u8   uint8_t
#define u16  uint16_t
#define u32  uint32_t
#define u64  uint64_t

#define i8   int8_t
#define i16  int16_t
#define i32  int32_t
#define i64  int64_t

#define f32  float
#define f64  double



#ifndef ARRAYSIZE
#define ARRAYSIZE(array) (sizeof(array) / sizeof((array)[0]))
#endif
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLAMP(x, low, high) (MAX((low), MIN((x), (high))))
#define SWAP(a, b) do { __typeof__(a) tmp = a; a = b; b = tmp; } while (0)


// 🐞 Debugging Helper
#ifdef DEBUG
#include <stdio.h>
#define DBG(fmt, ...) \
    fprintf(stderr, "[%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define DBG(fmt, ...)
#endif

#if COMPILER_MSVC
# define AlignOf(T) __alignof(T)
#elif COMPILER_CLANG
# define AlignOf(T) __alignof(T)
#elif COMPILER_GCC
# define AlignOf(T) __alignof__(T)
#else
// errror not defined 
#endif

#define MEMBER_OFFSET(Type, Member) (offsetof(Type, Member))
#define POINTER_FROM_MEMBER_OFFSET(MemberType, BasePtr, Offset) \
    ((MemberType *)((unsigned char *)(BasePtr) + (Offset)))

 #define DUMMY_MEMBER_PTR(Type, Member) (((Type*)0)->Member)
 
 #define CONTAINER_FROM_MEMBER(MemberPtr, ContainerType, Member) \
    ((ContainerType *)((unsigned char *)(MemberPtr) - MEMBER_OFFSET(ContainerType, Member)))


#define EachElement(it, array) (U64 it = 0; it < ARRAYSIZE(array); it += 1)
#define EachElementReverse(it, array) (U64 it = ARRAYSIZE(array) - 1; it < ARRAYSIZE(array) && it != (U64)-1; it -= 1)
#define EachElementWithValue(it, array) \
    for (U64 it = 0; it < ARRAYSIZE(array); it += 1) \
        if (true) { auto &value = array[it]; (void)value;}

 #define WITH_SETUP_AND_CLEANUP_BLOCK(setup_code, cleanup_code) \
    for (int _defer_loop_flag = ((setup_code), 0); !_defer_loop_flag; _defer_loop_flag += 1, (cleanup_code))

#define WITH_OPTIONAL_SETUP_AND_CLEANUP_BLOCK(setup_result, cleanup_code) \
    for (int _defer_flag = 2 * !(setup_result); (_defer_flag == 2 ? ((cleanup_code), 0) : !_defer_flag); _defer_flag += 1, (cleanup_code))

    #define FOR_EACH_INDEX(index_var, count) \
    for (size_t index_var = 0; index_var < (count); ++index_var)

    #define FOR_EACH_ELEMENT_INDEX_IN_ARRAY(index_var, array) \
    for (size_t index_var = 0; index_var < (sizeof(array)/sizeof((array)[0])); ++index_var)
#define FOR_EACH_ENUM_VALUE(enum_type, enum_var) \
    for (enum_type enum_var = (enum_type)0; enum_var < enum_type##_COUNT; enum_var = (enum_type)(enum_var + 1))
#define FOR_EACH_NONZERO_ENUM_VALUE(enum_type, enum_var) \
    for (enum_type enum_var = (enum_type)1; enum_var < enum_type##_COUNT; enum_var = (enum_type)(enum_var + 1))

       



#endif 
