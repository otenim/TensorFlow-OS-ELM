# OS-ELM-with-Python

## Environments

* Python==3.5.2
* Numpy==1.13.1

## Experimental Result

### MNIST

* input features: 784
* output features: 10
* training images: 60000
* validation images: 10000

**Best setting(val acc)**

* Batch size: 64
* Hidden unit: 1024
* val acc: 93.705%
* time: 0.042997sec

#### Batch size: 32
<table>
    <tr>
        <th>Hidden units</th>
        <th>time [sec]</th>
        <th>val loss</th>
        <th>val acc [%]</th>
    </tr>
    <tr>
        <td>128</td>
        <td>0.002255</td>
        <td>0.215616</td>
        <td>83.594800</td>
    </tr>
    <tr>
        <td>256</td>
        <td>0.004132</td>
        <td>0.177654</td>
        <td>88.192400</td>
    </tr>
    <tr>
        <td>512</td>
        <td>0.011603</td>
        <td>0.144838</td>
        <td>91.310200</td>
    </tr>
    <tr>
        <td>1024</td>
        <td>0.045471</td>
        <td>0.116035</td>
        <td>93.651400</td>
    </tr>
</table>

#### Batch size: 64
<table>
<tr>
    <th>Hidden units</th>
    <th>time [sec]</th>
    <th>val loss</th>
    <th>val acc [%]</th>
</tr>
<tr>
        <td>128</td>
        <td>0.004536</td>
        <td>0.214952</td>
        <td>83.791400</td>
</tr>
<tr>
        <td>256</td>
        <td>0.006333</td>
        <td>0.177618</td>
        <td>88.163600</td>
</tr>
<tr>
        <td>512</td>
        <td>0.012538</td>
        <td>0.145117</td>
        <td>91.239000</td>
</tr>
<tr>
        <td>1024</td>
        <td>0.042997</td>
        <td>0.115728</td>
        <td>93.705600</td>
</tr>
</table>

#### Batch size: 128
<table>
<tr>
    <th>Hidden units</th>
    <th>time [sec]</th>
    <th>val loss</th>
    <th>val acc [%]</th>
</tr>
<tr>
        <td>128</td>
        <td>0.014065</td>
        <td>0.215475</td>
        <td>83.674200</td>
</tr>
<tr>
        <td>256</td>
        <td>0.016116</td>
        <td>0.177782</td>
        <td>88.164400</td>
</tr>
<tr>
        <td>512</td>
        <td>0.023775</td>
        <td>0.144445</td>
        <td>91.275000</td>
</tr>
<tr>
        <td>1024</td>
        <td>0.057628</td>
        <td>0.115973</td>
        <td>93.665000</td>
</tr>
</table>

#### Batch size: 256
<table>
<tr>
    <th>Hidden units</th>
    <th>time [sec]</th>
    <th>val loss</th>
    <th>val acc [%]</th>
</tr>
<tr>
        <td>128</td>
        <td>0.033858</td>
        <td>0.215426</td>
        <td>83.608400</td>
</tr>
<tr>
        <td>256</td>
        <td>0.049971</td>
        <td>0.177684</td>
        <td>88.226400</td>
</tr>
<tr>
        <td>512</td>
        <td>0.061894</td>
        <td>0.144741</td>
        <td>91.290600</td>
</tr>
<tr>
        <td>1024</td>
        <td>0.103169</td>
        <td>0.115896</td>
        <td>93.654000</td>
</tr>
</table>

#### Batch size: 512
<table>
<tr>
    <th>Hidden units</th>
    <th>time [sec]</th>
    <th>val loss</th>
    <th>val acc [%]</th>
</tr>
<tr>
        <td>128</td>
        <td>0.126693</td>
        <td>0.215809</td>
        <td>83.709400</td>
</tr>
<tr>
        <td>256</td>
        <td>0.139817</td>
        <td>0.177832</td>
        <td>88.148000</td>
</tr>
<tr>
        <td>512</td>
        <td>0.212566</td>
        <td>0.144978</td>
        <td>91.271400</td>
</tr>
<tr>
        <td>1024</td>
        <td>0.272930</td>
        <td>0.115837</td>
        <td>93.629000</td>
</tr>
</table>

### Digits

* input features: 64
* output features: 10
* training images: 1437
* validation images: 360

**Best setting(val acc)**

* Batch size: 64
* Hidden units: 512
* val acc: 95.13%
* time: 0.011173sec

#### Batch size: 8
<table>
<tr>
    <th>Hidden units</th>
    <th>time [sec]</th>
    <th>val loss</th>
    <th>val acc [%]</th>
</tr>
<tr>
        <td>64</td>
        <td>0.000309</td>
        <td>0.179802</td>
        <td>87.750000</td>
</tr>
<tr>
        <td>128</td>
        <td>0.000627</td>
        <td>0.140495</td>
        <td>91.527778</td>
</tr>
<tr>
        <td>256</td>
        <td>0.001634</td>
        <td>0.113753</td>
        <td>93.850000</td>
</tr>
<tr>
        <td>512</td>
        <td>0.005928</td>
        <td>0.100869</td>
        <td>94.833333</td>
</tr>
<tr>
        <td>1024</td>
        <td>0.029787</td>
        <td>0.140027</td>
        <td>94.233333</td>
</tr>
</table>

#### Batch size: 16
<table>
<tr>
    <th>Hidden units</th>
    <th>time [sec]</th>
    <th>val loss</th>
    <th>val acc [%]</th>
</tr>
<tr>
        <td>64</td>
        <td>0.000608</td>
        <td>0.181048</td>
        <td>87.588889</td>
</tr>
<tr>
        <td>128</td>
        <td>0.001061</td>
        <td>0.140012</td>
        <td>91.105556</td>
</tr>
<tr>
        <td>256</td>
        <td>0.002227</td>
        <td>0.113563</td>
        <td>93.855556</td>
</tr>
<tr>
        <td>512</td>
        <td>0.007483</td>
        <td>0.101347</td>
        <td>94.900000</td>
</tr>
<tr>
        <td>1024</td>
        <td>0.033968</td>
        <td>0.140204</td>
        <td>93.905556</td>
</tr>
</table>

#### Batch size: 32
<table>
<tr>
    <th>Hidden units</th>
    <th>time [sec]</th>
    <th>val loss</th>
    <th>val acc [%]</th>
</tr>
<tr>
        <td>64</td>
        <td>0.001269</td>
        <td>0.180502</td>
        <td>87.522222</td>
</tr>
<tr>
        <td>128</td>
        <td>0.001865</td>
        <td>0.140594</td>
        <td>91.233333</td>
</tr>
<tr>
        <td>256</td>
        <td>0.003250</td>
        <td>0.114752</td>
        <td>93.600000</td>
</tr>
<tr>
        <td>512</td>
        <td>0.009370</td>
        <td>0.100424</td>
        <td>95.005556</td>
</tr>
<tr>
        <td>1024</td>
        <td>0.039985</td>
        <td>0.140275</td>
        <td>94.116667</td>
</tr>
</table>

#### Batch size: 64
<table>
<tr>
    <th>Hidden units</th>
    <th>time [sec]</th>
    <th>val loss</th>
    <th>val acc [%]</th>
</tr>
<tr>
        <td>64</td>
        <td>0.003376</td>
        <td>0.181260</td>
        <td>87.433333</td>
</tr>
<tr>
        <td>128</td>
        <td>0.004053</td>
        <td>0.143445</td>
        <td>91.122222</td>
</tr>
<tr>
        <td>256</td>
        <td>0.005623</td>
        <td>0.113962</td>
        <td>93.527778</td>
</tr>
<tr>
        <td>512</td>
        <td>0.011173</td>
        <td>0.100105</td>
        <td>95.133333</td>
</tr>
<tr>
        <td>1024</td>
        <td>0.038069</td>
        <td>0.139524</td>
        <td>94.083333</td>
</tr>
</table>

#### Batch size: 128
<table>
<tr>
    <th>Hidden units</th>
    <th>time [sec]</th>
    <th>val loss</th>
    <th>val acc [%]</th>
</tr>
<tr>
        <td>64</td>
        <td>0.008676</td>
        <td>0.179460</td>
        <td>87.566667</td>
</tr>
<tr>
        <td>128</td>
        <td>0.011528</td>
        <td>0.142623</td>
        <td>91.177778</td>
</tr>
<tr>
        <td>256</td>
        <td>0.014602</td>
        <td>0.114893</td>
        <td>93.683333</td>
</tr>
<tr>
        <td>512</td>
        <td>0.019343</td>
        <td>0.100079</td>
        <td>95.116667</td>
</tr>
<tr>
        <td>1024</td>
        <td>0.052509</td>
        <td>0.139227</td>
        <td>94.272222</td>
</tr>
</table>

#### Batch size: 256
<table>
<tr>
    <th>Hidden units</th>
    <th>time [sec]</th>
    <th>val loss</th>
    <th>val acc [%]</th>
</tr>
<tr>
        <td>64</td>
        <td>0.024384</td>
        <td>0.178305</td>
        <td>87.911111</td>
</tr>
<tr>
        <td>128</td>
        <td>0.028062</td>
        <td>0.141705</td>
        <td>91.127778</td>
</tr>
<tr>
        <td>256</td>
        <td>0.040147</td>
        <td>0.114101</td>
        <td>93.661111</td>
</tr>
<tr>
        <td>512</td>
        <td>0.045700</td>
        <td>0.101385</td>
        <td>95.011111</td>
</tr>
<tr>
        <td>1024</td>
        <td>0.074501</td>
        <td>0.140002</td>
        <td>94.283333</td>
</tr>
</table>

### Boston

* input features: 64
* output features: 10
* training images: 404
* validation images: 102

**Best Setting(val loss)**

* Batch size: 4
* Hidden units: 16
* val loss: 0.009228
* time: 0.000083sec

#### Batch size: 4
<table>
<tr>
    <th>Hidden units</th>
    <th>time [sec]</th>
    <th>val loss</th>
</tr>
<tr>
        <td>8</td>
        <td>0.000078</td>
        <td>0.010738</td>
</tr>
<tr>
        <td>16</td>
        <td>0.000083</td>
        <td>0.009228</td>
</tr>
<tr>
        <td>32</td>
        <td>0.000105</td>
        <td>0.010682</td>
</tr>
<tr>
        <td>64</td>
        <td>0.000195</td>
        <td>0.011566</td>
</tr>
<tr>
        <td>128</td>
        <td>0.000599</td>
        <td>0.015179</td>
</tr>
</table>

#### Batch size: 8
<table>
<tr>
    <th>Hidden units</th>
    <th>time [sec]</th>
    <th>val loss</th>
</tr>
<tr>
        <td>8</td>
        <td>0.000116</td>
        <td>0.010738</td>
</tr>
<tr>
        <td>16</td>
        <td>0.000118</td>
        <td>0.009228</td>
</tr>
<tr>
        <td>32</td>
        <td>0.000138</td>
        <td>0.010682</td>
</tr>
<tr>
        <td>64</td>
        <td>0.000262</td>
        <td>0.011566</td>
</tr>
<tr>
        <td>128</td>
        <td>0.000640</td>
        <td>0.015179</td>
</tr>

</table>

#### Batch size: 16
<table>
<tr>
    <th>Hidden units</th>
    <th>time [sec]</th>
    <th>val loss</th>
</tr>
<tr>
        <td>8</td>
        <td>0.000161</td>
        <td>0.010738</td>
</tr>
<tr>
        <td>16</td>
        <td>0.000250</td>
        <td>0.009228</td>
</tr>
<tr>
        <td>32</td>
        <td>0.000289</td>
        <td>0.010682</td>
</tr>
<tr>
        <td>64</td>
        <td>0.000448</td>
        <td>0.011566</td>
</tr>
<tr>
        <td>128</td>
        <td>0.000864</td>
        <td>0.015179</td>
</tr>
</table>

#### Batch size: 32
<table>
<tr>
    <th>Hidden units</th>
    <th>time [sec]</th>
    <th>val loss</th>
</tr>
<tr>
        <td>8</td>
        <td>0.000257</td>
        <td>0.010738</td>
</tr>
<tr>
        <td>16</td>
        <td>0.000390</td>
        <td>0.009228</td>
</tr>
<tr>
        <td>32</td>
        <td>0.000597</td>
        <td>0.010682</td>
</tr>
<tr>
        <td>64</td>
        <td>0.001074</td>
        <td>0.011566</td>
</tr>
<tr>
        <td>128</td>
        <td>0.001506</td>
        <td>0.015179</td>
</tr>
</table>

#### Batch size: 64
<table>
<tr>
    <th>Hidden units</th>
    <th>time [sec]</th>
    <th>val loss</th>
</tr>
<tr>
        <td>8</td>
        <td>0.001130</td>
        <td>0.010738</td>
</tr>
<tr>
        <td>16</td>
        <td>0.001287</td>
        <td>0.009228</td>
</tr>
<tr>
        <td>32</td>
        <td>0.002012</td>
        <td>0.010682</td>
</tr>
<tr>
        <td>64</td>
        <td>0.002544</td>
        <td>0.011566</td>
</tr>
<tr>
        <td>128</td>
        <td>0.003602</td>
        <td>0.015179</td>
</tr>
</table>
