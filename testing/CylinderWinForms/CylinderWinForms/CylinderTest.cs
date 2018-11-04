using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;     // DLL support

namespace CylinderWinForms
{
    



    [StructLayout(LayoutKind.Sequential)]
    class CylinderTest
    {


        public const string LibName = "CylinderComponent.dll";

        public const string absPath = "C:\\Users\\Andy\\src\\mx-build\\testing\\CylinderComponent\\Debug\\CylinderComponent.dll";


        public const string modelPath = @"C:\Users\Andy\src\mechanica\testing\models\hex_cylinder.1.obj";


        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        static extern bool SetDllDirectory(string lpPathName);

        [DllImport("kernel32.dll", SetLastError = true)]
        static extern IntPtr LoadLibrary(string lpFileName);

        static CylinderTest()
        {
            bool result = SetDllDirectory("C:\\Users\\Andy\\src\\mx-build\\testing\\CylinderComponent\\Debug");

            System.Console.WriteLine("set dll dir: " + result);

            IntPtr lib = LoadLibrary(LibName);

            System.Console.WriteLine("lib: " + lib);
            Console.WriteLine(Marshal.GetLastWin32Error());

            lib = LoadLibrary(absPath);

            System.Console.WriteLine("lib: " + lib);
            Console.WriteLine(Marshal.GetLastWin32Error());
        }


        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern Int32 CylinderTest_Create(Int32 width, Int32 height, out IntPtr result);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern Int32 CylinderTest_Draw(IntPtr comp);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern Int32 CylinderTest_Step(IntPtr comp, float step);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern Int32 CylinderTest_LoadMesh(IntPtr comp, [MarshalAs(UnmanagedType.LPStr)] String s);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern Int32 CylinderTest_GetScalarValue(IntPtr comp, UInt32 id, [Out]float result);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern Int32 CylinderTest_SetScalarValue(IntPtr comp, UInt32 id, float value);

   


    }


}
