# OS-ELM-with-Python

## Environments

* Python==3.5.2
* Numpy==1.13.1

## Experimental Result

### MNIST

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
