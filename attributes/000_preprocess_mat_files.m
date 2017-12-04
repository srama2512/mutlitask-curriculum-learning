load lfw_att_40.mat

outstruct = struct();
outstruct.AttrName = AttrName;
outstruct.label = label;
outstruct.name = name;

for i = 1:length(outstruct.name)
    curr_name = outstruct.name(i);
    curr_name{1} = strrep(curr_name{1}, '\', '/');
    outstruct.name(i) = curr_name;
end
    
savejson('', outstruct, 'annotations.json');