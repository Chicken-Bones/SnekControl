﻿using System;
using System.Windows.Controls;

namespace SnekControl
{
    /// <summary>
    /// Interaction logic for MotorControl.xaml
    /// </summary>
    public partial class MotorControl : UserControl
    {
	    public MotorControl()
        {
            InitializeComponent();
        }

		internal void Command(string text) {
			throw new NotImplementedException();
		}
	}
}
