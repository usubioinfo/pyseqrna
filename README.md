pySeqRNA
========

pySeqRNA: a python-based package for RNASeq data analysis

Today, massive amounts of data are generated by Next-Generation Sequencing (NGS) technologies. In recent years, many algorithms, statistical methods, and software tools have been developed to perform the individual analysis steps of various NGS applications. However, streamlined analysis remains a significant barrier to effectively utilizing the technology. We have developed a Python package (pySeqRNA) that allows fast, efficient, manageable, and reproducible RNA-Seq analysis. It effectively uses current software and tools with newly written Python scripts without confining users to a collection of pre-defined methods and environments by combining many command-line tools and custom Python scripts.

Input
-----

PySeqRNA requires a input file containing information of samples and input read files. Input template and example files here:

|\# Project title/Information lines should start with \#|||||
| :--- | :--- | :--- | :--- | :--- |
| SampleName | Replication | Identifier | File1 | File2 |
| AddFull Sample Name Here | Add Replication Here | Add sample Identifier Here | Add Sample File Name Here | Add Reverese File here if Paired END |

Example input file:

|\#Arabidopsis transcriptome study under high light stress|||||
| :---: | :---: | :---: | :---: | :---: |
| SampleName | Replication | Identifier | File1 | File2 |
| GL0.5h1 | GL0.5h1 | GL0.5 | SRR6767632_001.fastq.gz | SRR6767632_002.fastq.gz |
| GLO.5h2 | GLO.5h2 | GL0.5 | SRR6767633_001.fastq.gz | SRR6767633_002.fastq.gz |
| GL6h1 | GL6h1 | GL6 | SRR6767634_001.fastq.gz | SRR6767634_002.fastq.gz |
| GL6h2 | GL6h2 | GL6 | SRR6767635_001.fastq.gz | SRR6767635_002.fastq.gz |
| GL12h1 | GL12h1 | GL12 | SRR6767636_001.fastq.gz | SRR6767636_002.fastq.gz |
| GL12h2 | GL12h2 | GL12 | SRR6767637_001.fastq.gz | SRR6767637_002.fastq.gz |
| GL24h1 | GL24h1 | GL24 | SRR6767639_001.fastq.gz | SRR6767639_002.fastq.gz |
| GL24h2 | GL24h2 | GL24 | SRR6767640_001.fastq.gz | SRR6767640_002.fastq.gz |
| GL48h1 | GL48h1 | GL48 | SRR6767642_001.fastq.gz | SRR6767642_002.fastq.gz |
| GL48h2 | GL48h2 | GL48 | SRR6767643_001.fastq.gz | SRR6767643_002.fastq.gz |
| GL72h1 | GL72h1 | GL72 | SRR6767644_001.fastq.gz | SRR6767644_002.fastq.gz |
| GL72h2 | GL72h2 | GL72 | SRR6767645_001.fastq.gz | SRR6767645_002.fastq.gz |

Analysis approach
-----------------

The pySeqRNA perform RNA-Seq analysis in two steps: 

1. Uniquely mapped reads
2. Multimapped reads

Development Environment and Prerequisite
----------------------------------------

This source code was developed in Linux, and has been tested on Linux and OS X. The main prerequisite is Python > 3.7. Following are the external dependencies:

- Flexbar – flexible barcode and adapter removal [https://github.com/seqan/flexbar](https://github.com/seqan/flexbar)
- Trimmomatic: A flexible read trimming tool for Illumina NGS data [http://www.usadellab.org/cms/?page=trimmomatic](http://www.usadellab.org/cms/?page=trimmomatic)
- Trim Galore [https://github.com/FelixKrueger/TrimGalore](https://github.com/FelixKrueger/TrimGalore)
- SortMeRNA [https://github.com/sortmerna/sortmerna] (https://github.com/sortmerna/sortmerna)
- STAR Aligner [https://github.com/alexdobin/STAR](https://github.com/alexdobin/STAR)
- HISAT2 [http://daehwankimlab.github.io/hisat2/](http://daehwankimlab.github.io/hisat2/)
- Bowtie2 [https://github.com/BenLangmead/bowtie2](https://github.com/BenLangmead/bowtie2)
- Subread [https://subread.sourceforge.net/](https://subread.sourceforge.net/)
- HTSeq [https://github.com/simon-anders/htseq](https://github.com/simon-anders/htseq)
- Samtools [https://github.com/samtools/samtools](https://github.com/samtools/samtools)
- Bamtools [https://github.com/pezmaster31/bamtools](https://github.com/pezmaster31/bamtools)
- R Language [https://cran.r-project.org/bin/windows/base/](https://cran.r-project.org/bin/windows/base/)
- DESeq2 [https://bioconductor.org/packages/release/bioc/html/DESeq2.html](https://bioconductor.org/packages/release/bioc/html/DESeq2.html)
- edgeR [https://bioconductor.org/packages/release/bioc/html/edgeR.html](https://bioconductor.org/packages/release/bioc/html/edgeR.html)
- Python 3 [https://www.python.org/downloads/](https://www.python.org/downloads/)

Installation
------------

The installation of pySeqRNA can be done in two ways:

1. Create a dedicated miniconda3 environment
  
    Download pySeqRNA 0.2 from:
          
    [https://bioinfo.usu.edu/pyseqrna/download/pySeqRNA-0.2.tar.gz](https://bioinfo.usu.edu/pyseqrna/download/pySeqRNA-0.2.tar.gz)

    Download the Miniconda installer:
          
    [https://docs.conda.io/en/latest/miniconda.html#linux-installers](https://docs.conda.io/en/latest/miniconda.html#linux-installers)

    Extract the downloaded file:

    tar -xvzf pySeqRNA-0.2.tar.gz

    cd pySeqRNA-0.2

    chmod 755 INSTALL

    ./INSTALL
  
2. Create a docker image from docker file for cross-platform 

      
    Download pySeqRNA 0.2 from:
          
    [https://bioinfo.usu.edu/pyseqrna/download/pySeqRNA-0.2.tar.gz](https://bioinfo.usu.edu/pyseqrna/download/pySeqRNA-0.2.tar.gz)

    Extract the downloaded file:

    tar -xvzf pySeqRNA-0.2.tar.gz

    cd pySeqRNA-0.2

    docker build -t pyseqrna .
    
    

Run pyseqrna
------------

pyseqrna -h

Please Cite
-----------

Duhan N and Kaundal R. pySeqRNA: an automated Python package for RNA sequencing data analysis [version 1; not peer reviewed]. F1000Research 2020, 9(ISCB Comm J):1128 (poster) (https://doi.org/10.7490/f1000research.1118314.1) 

Queries and Contact
----------------------

Written by Naveen Duhan (naveen.duhan@usu.edu),

Kaundal Bioinformatics Lab, Utah State University,

Released under the terms of GNU General Public Licence v3

In case of technical problems (bugs etc.) please contact Naveen Duhan (naveen.duhan@usu.edu)

For any Questions on the scientific aspects of the pySeqRNA-0.2 method please contact:

Rakesh Kaundal, (rkaundal@usu.edu)

Naveen Duhan, (naveen.duhan@outlook.com)