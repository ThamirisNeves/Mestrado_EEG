#!/bin/bash
#####################################################################
# SCRIPT PARA RODAR A EEGNET PARA CADA SUJEITO UTILIZANDO O         #
# O ALGORITMO AMUSE (BASELINE DO TRABALHO DE MESTRADO)              #
#####################################################################

for i in $(seq 1 9)
    do 
    #entra no diretorio que se encontra o arquivo 
    cd '/Documents/Projeto_mestrado/Projeto_mestrado/Codigos/' 
    #muda valor global no codigo
    sed 's/XX/'$i'/g' teste_1_sobi.py > SOBI_classe_sujeito_$i/SOBI-EEGnet_usuario_$i.py 
    #entra na pasta
    cd SOBI_classe_sujeito_$i 
    #roda o programa
    /bin/python3 SOBI-EEGnet_usuario_$i.py
    #sai da pasta
    cd ..
    done 
    
