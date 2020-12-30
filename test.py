def map_id_to_song(self, file):
        song_dict = {}
        original_file = pd.read_csv('.\data.csv')
        with open(file,encoding='utf-16') as src:
            for line in src:
                input_song, output_songs = line.split('\t')
                output_songs = output_songs.replace('\n', '')
                song_dict[input_song] = output_songs
        
        for song in song_dict:
            
            for suggested_song in song:
                df_new = df_new.append(original_file[(original_file['id'] == suggested_song)])



map_id_to_song('.\output.json')