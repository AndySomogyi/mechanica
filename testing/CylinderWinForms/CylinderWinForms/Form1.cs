﻿using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace CylinderWinForms
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            IntPtr obj = IntPtr.Zero;
            CylinderTest.CylinderTest_Create(600, 900, out obj);

            CylinderTest.CylinderTest_LoadMesh(obj, CylinderTest.modelPath);
        }
    }
}
